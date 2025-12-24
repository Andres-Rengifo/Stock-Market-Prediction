public class testing {
    class Node {
        int val;
        Node next;
        Node(int val) { this.val = val; }
    }

    public static void printList(Node head) {
        Node current = head;
        while(current != null) {
            System.out.println(current.val);
            current = current.next;
        }
    }
    public static void main(String[] args) {
        Node head = new testing().new Node(1);
        head.next = new testing().new Node(2);
        head.next.next = new testing().new Node(3);

        printList(head);
    }
}

